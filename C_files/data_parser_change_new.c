// OUtput <476>
// last timestape: "23923.00"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <windows.h>
#include <string.h>
#include <wctype.h>
#include <signal.h>

#define COL_NUM 7
#define CAR_COL 1
#define SECT_COL 6
#define TIME_COL 0
#define BIRTH_INTERVAL 250 //time unit is millisecond.
/*
#define X_START 4800 //leftest
#define X_END 8800 //rightest
#define Y_START 9000 //top
#define Y_END 5000 //bottom
*/
#define CORD_MAX 2147483647
#define RSU_NUM 8
#define RSU_RAD 500 //unit is meter.

int X_START = 0; //leftest
int X_END = 0; //rightest
int Y_START = 0; //top
int Y_END = 0; //bottom

int RSU_CORD[RSU_NUM][2]; //RSU_CORD[][0] for x, RSU_CORD[][1] for y.

struct linked_list
{
	int id;
	int val;
	double x;
	double y;
	char name[64];
	struct linked_list *second_list;
	struct linked_list *second_list_tail;
	struct linked_list *next;
};

char *getWord(char *input, char *string); //put the word between two commas in 'input' into the argument 'string'; return the current 'input' ptr.
struct linked_list *listInsert(struct linked_list *tail, int id); //insert a new node; return the tail.
void freeList(struct linked_list *head); //release the memory of a linked list.
struct linked_list *findListId(struct linked_list *head, int target_id); //find out wether the list contains the node with targer_id; return list ptr if the target_id exit, else return NULL.
struct linked_list *deleteNode(struct linked_list *head, struct linked_list *delete_node); //delete the node in the linked list.
void carListTraverse(struct linked_list *car_head, int msg_num, FILE *fl_ptr); //traverse the car lists and display their elements into 'fl_ptr' file.
void listTraverse(struct linked_list *head); //traverse the linked list and display their elements.
int getListSize(struct linked_list *head); //return the size of a linked list.
struct linked_list *getAllSect(struct linked_list *car_list); //copy all the sections into a new section list(without duplicate) from 'car_list', and return the section list's head.
void outputData(int car_num, int msg_num, int t0, int rb_num, struct linked_list *car_list, struct linked_list *sect_list, FILE *fl_ptr, int time, int v_size); //output the suitable data for experiment into the 'fl_ptr'.
int IArrSearch(int *arr, int arr_size, int target); //search the target in the integer array, and return it's index.
struct linked_list *getListTail(struct linked_list *head); //return the tail of that linked list.
int vCntrCheck(struct linked_list *car_list, int car_num, struct linked_list *sect_list, int msg_num); //check wether the repeatedly received number is correct. return 0 if it's not correct.
int findMedVal(struct linked_list *sect_list, int msg_num); //find and return the median val in the linked list.
int valCmp(const void *val1, const void *val2); //qsort cmp function.
void printList(struct linked_list *head); //return the size of a linked list.
int ListSearch(struct linked_list *head, int target); //return the size of a linked list.
struct linked_list *ChangeInToBig(struct linked_list *sect_list);
int GetBigMap(int i);
int listRangeFindId(struct linked_list* head, struct linked_list* end, int traget_id);
void *ListForFlag(struct linked_list *veh_list, struct linked_list *flag_list_head, struct linked_list *flag_list_tail);


int toVehNumTime(FILE *fl_ptr, int veh_num); //let the file pointer points to the time which has vehicle number bigger than 'vch_num'. Return 0 if failed.
char *parseVehName(char *line, char *veh_name); //parse the vehicle's name from a xml line. Return the name if success, else NULL.
double parseVehX(char *line); //parse the vehicle's x coordinate. Return CORD_MAX if failed.
double parseVehY(char *line); //parse the vehicle's y coordinate. Return CORD_MAX if failed.
struct linked_list *getVehInTXY(FILE *fl_ptr); //parse the vehicles in (X_START, Y_START) ~ (X_END, Y_END) at a time step, i.e., parse until meet </timestep>, from 'fl_ptr', and record the passed sections as well. Return the linked list's head of the vehicles if success, else NULL.
struct linked_list *listNameInsert(struct linked_list **head, struct linked_list *tail, char *name); //insert a new node with name. Return the tail.
struct linked_list *getVehSect(struct linked_list *veh_list); //extract all the sections vehicles passed. Return the head of the section. 
struct linked_list *listIdInsert(struct linked_list **head, struct linked_list *tail, int id); //insert a new node with id. Return the tail.
int listNameErrCheck(struct linked_list *head); //check whether the name is duplicate.
int listIdErrCheck(struct linked_list *head); //check whether the id is duplicate.
void doubleListDisplay(struct linked_list *list_head); //print the content in a linked list doubly.
struct linked_list *parseVehSect(FILE *fl_ptr, struct linked_list *veh_head, int time); //parse the section that the vehicles will pass in next 'time' seconds in the 'fl_ptr' file.
char *toNextTimeStep(FILE *fl_ptr, char *line); //move file pointer to the beginning of next time step. Return the next time stpe line.
struct linked_list *findListName(struct linked_list *head, char *name); //find the 'name' in the list. Return the list if success, else NULL.
struct linked_list *getVehInTRSU(FILE *fl_ptr, int flag); //parse the vehicles in RSU covered range at a time step, i.e., parse until meet </timestep>, from 'fl_ptr', and record the passed sections as well. Return the linked list's head of the vehicles if success, else NULL.
struct linked_list *filterList(struct linked_list *head, int target_size); //filter the list randomly into 'target_size', if the list size is already lesser than 'target_size' , then it won't be filtered. Return the list.
struct linked_list *getIdxNode(struct linked_list *head, int idx); //get the node that index is 'idx' in the list. Return the list ptr if success, else NULL.

FILE *fl_input = NULL, *fl_output = NULL, *fl_output_map = NULL;
void setRSUCORD(); //set the coordinate of the RSUs by reading file RSU_CORD.txt.
char time_now[50] = "";
int last_time = 0;
int time_val = 0;

static int iter_ = 0;

void signal_handler(int signum) {
    printf("signal_handler: caught signal %d\n", signum);
    if (signum == SIGINT) {
        printf("SIGINT\n");	
		fclose(fl_input);
		fclose(fl_output);
        exit(1);
    }
}

struct linked_list *flag_list = NULL, *flag_list_tail = NULL, *flag_list_buf = NULL;



int main(int argc, char **argv)
{
	char file_line_buf[1024] = "\0", word[64] = "\0";
	char *ptr = NULL;
	int col_num_cntr = 0;
	int id_buf = 0;
	int str_len = 0;
	int start_time = 0, current_time = 0, t0 = 0;
	int car_num = 0, msg_num = 0;
	int car_cntr = 0, msg_cntr = 0;
	int time = 0;
	int v_size = 0;
	struct linked_list *car_list = NULL, *car_list_tail = NULL, *car_list_buf = NULL;
	struct linked_list *sect_list = NULL, *sect_list_tail = NULL, *sect_list_buf = NULL;
	// flag_list = (struct linked_list *) malloc (sizeof(struct linked_list) * 1);

	if (signal(SIGINT, signal_handler) == SIG_ERR) {
    	printf("Failed to caught signal\n");
    }

		fl_input = fopen(*(argv + 1), "r"); //open input & output files.
		if(fl_input == NULL)
		{
			fprintf(stderr, "!!!!!	Dataset file with Path	%s	is failed to open. !!!!!\n", *(argv + 1));
			exit(1);
		}
		fl_output = fopen(*(argv + 2), "w+");
		if(fl_output == NULL)
		{
			fprintf(stderr, "!!!!!	WARNING: Output file with Path	%s	is failed to open. !!!!!\n", *(argv + 2));
			fl_output = stdout;
		}
		fl_output_map = fopen(*(argv + 3), "w+");
		if(fl_output_map == NULL)
		{
			fprintf(stderr, "!!!!!	WARNING: Output file with Path	%s	is failed to open. !!!!!\n", *(argv + 2));
			fl_output_map = stdout;
		}

		printf("Enter the total car number should be covered: ");
		scanf("%d", &car_num); //get the vehicle number for range searching.
		printf("Enter the tracking time: ");
		scanf("%d", &time); //get the time interval we should parse.
		printf("Enter the required vehicle number: ");
		scanf("%d", &v_size); //get the required number of the vehicles.
		printf("Enter the last time: ");
		scanf("%d", &last_time); //get the last time.
		time = time -1;
		
		while(toVehNumTime(fl_input, car_num) == 0)
		{
			fprintf(stderr, "!!!!! There is not enough vehicle in the file. ReEnter again. !!!!!\n");
			fseek(fl_input, 0, SEEK_SET);
			scanf("%d", &car_num); //get the required vehicle number.
		}
		while(time < 1)
		{
			fprintf(stderr, "!!!!! Parsing time interval can't be smaller than 1. ReEnter again. !!!!!\n");
			scanf("%d", &time); //get the time interval we should parse.
		}
		while(v_size < 0)
		{
			fprintf(stderr, "!!!!! Required number of the vehicles should be positive. ReEnter aagain. !!!!!\n");
			scanf("%d", &v_size);
		}

		setRSUCORD(); //設定座標

		fgets(file_line_buf, 1024, fl_input); //skip the <timesetp ...>.
		//printf("%s", file_line_buf);
		/*
		for(int i = 0; i < car_num; i ++)
		{
			fgets(file_line_buf, 1024, fl_input);
			printf("name: %s\tx:%lf\ty:%lf\n", parseVehName(file_line_buf, word), parseVehX(file_line_buf), parseVehY(file_line_buf));
		}*/


		for(int i = 0;i<200;i++){
			//car_list = getVehInTXY(fl_input); //construct the linked list of the vehicles.
			car_list = getVehInTRSU(fl_input, i); //construct the linked list of the vehicles.
			car_list = filterList(car_list, v_size); //filter vehicles into the required size.
			car_list = parseVehSect(fl_input, car_list, time);
			sect_list = getVehSect(car_list); //construct the linked list of the section which the vehicles passed.
			//listTraverse(car_list);
			//listTraverse(sect_list);
			//doubleListDisplay(car_list); //display the vehicles' path and info.
			outputData(getListSize(car_list), getListSize(sect_list), 1000, getListSize(sect_list), car_list, sect_list, fl_output, time, v_size);
			last_time = time_val;
			// printf("%d\n", getListSize(car_list));
		}

	fclose(fl_input);
	fclose(fl_output);
	fclose(fl_output_map);
	freeList(car_list);
	freeList(sect_list);

	return 0;
}

char *getWord(char *input, char *word)
{
	int lenght = 0;
	char *in_ptr = NULL, *w_ptr = NULL, *end = NULL;

	lenght = strlen(input);
	in_ptr = input;
	w_ptr = word;
	end = input + lenght;

	while(in_ptr != NULL && iswspace(*in_ptr) && in_ptr < end)
	{
		in_ptr ++;
	}
	while(in_ptr != NULL && in_ptr < end)
	{
		if(*in_ptr == ',')
		{
			in_ptr ++;
			break;
		}
		if(*in_ptr == '\n')
		{
			in_ptr = NULL;
			break;
		}
		*w_ptr ++ = *in_ptr ++;
	}
	*w_ptr = ',';
	*(w_ptr + 1) = '\0';

	return in_ptr;
}

struct linked_list *listInsert(struct linked_list *tail, int id) //insert a new node; return the tail.
{
	if(tail != NULL)
	{
		tail->next = (struct linked_list *) malloc(sizeof(struct linked_list) * 1);
		tail->next->id = id;
		tail->next->val = 0;
		tail->next->x = 0.0;
		tail->next->y = 0.0;
		tail->next->name[0] = '\0';
		tail->next->second_list = NULL;
		tail->next->second_list_tail = NULL;
		tail->next->next = NULL;
		tail = tail->next;
	}
	else
	{
		tail = (struct linked_list *) malloc(sizeof(struct linked_list) * 1);
		tail->id = id;
		tail->val = 0;
		tail->x = 0.0;
		tail->y = 0.0;
		tail->name[0] = '\0';
		tail->second_list = NULL;
		tail->second_list_tail = NULL;
		tail->next = NULL;
	}
	//printf("listInsert() is completed\n");

	return tail;
}

struct linked_list *findListId(struct linked_list *head, int target_id) //find out wether the list contains the node with target_id; return list ptr if the target_id exit, else return NULL.
{
	struct linked_list *linked_list_ptr = head;

	while(linked_list_ptr != NULL)
	{
		if(linked_list_ptr->id == target_id)
		{
			return linked_list_ptr;
		}
		
		linked_list_ptr = linked_list_ptr->next;
	}
	
	return NULL;
}

struct linked_list *deleteNode(struct linked_list *head, struct linked_list *target_node) //delete the node in the linked list.
{
	struct linked_list *linked_list_pre = head;
	struct linked_list *linked_list_ptr = head;
	int id_buf = 0;
	char name_buf[64] = "\0";

	if(head == NULL)
	{
		fprintf(stderr, "!!!!! In deleteNode(), the 'head' is NULL. !!!!!\n");
		exit(1);
	}
	else if(target_node == NULL)
	{
		fprintf(stderr, "!!!!! In deleteNode(), the 'target_node' is NULL. !!!!!\n");
		exit(1);
	}

	if(head == target_node)
	{
		head = linked_list_ptr->next;
		freeList(linked_list_ptr->second_list);
		free(linked_list_ptr);

		//printf("deleteNode() is completed; head node is deleted.\n");
		return head;
	}

	while(linked_list_ptr != NULL)
	{
		if(linked_list_ptr == target_node)
		{
			id_buf = target_node->id;
			strcpy(name_buf, target_node->name);
			linked_list_pre->next = linked_list_ptr->next;
			freeList(linked_list_ptr->second_list);
			free(linked_list_ptr);

			//printf("deleteNode() is completed; node with (id, name): (%d, %s) is deleted.\n", id_buf, name_buf);
			return head;
		}

		linked_list_pre = linked_list_ptr;
		linked_list_ptr = linked_list_ptr->next;
	}
	//printf("In deleteNode(), 'delete_node(id: %d, name: %s)' is not found\n", target_node->id, target_node->name);

	return head;
}

void freeList(struct linked_list *head) //release the memory of a linked list.
{
	struct linked_list *linked_list_ptr = head, *linked_list_pre = head;
	//struct linked_list *sec_linked_list_ptr = NULL, *sec_linked_list_pre = NULL;

	while(linked_list_ptr != NULL)
	{
		if(linked_list_ptr->second_list != NULL) //all variables should be initailized.
		{
			freeList(linked_list_ptr->second_list);
		}
		linked_list_pre = linked_list_ptr;
		linked_list_ptr = linked_list_ptr->next;
		linked_list_pre->second_list = NULL;
		linked_list_pre->second_list_tail = NULL;
		linked_list_pre->next =NULL;
		free(linked_list_pre);
	}
	//printf("freeList() is completed.\n");
}

void carListTraverse(struct linked_list *car_head, int msg_num, FILE *fl_ptr) //traverse the car lists and display their elements into 'fl_ptr' file.
{
	int *sect_mapping_tb = NULL; //map those unformated section id into 0~msg_num-1.
	int val_buf = 0, big_val_buff = 0;
	int list_size = 0;
	int org_big_target = 0;
	struct linked_list *linked_list_ptr = car_head, *linked_list_second_ptr = NULL;
	struct linked_list *sect_list = NULL, *sect_list_ptr = NULL;
	struct linked_list *Big_sect_list = NULL, *Big_sect_list_ptr = NULL;
	struct linked_list *Big_flag_list = NULL, *Big_flag_list_ptr = NULL;

	if(linked_list_ptr == NULL)
	{
		printf("In carListTraverse(), head is NULL. No node is printed\n");
	}
	else
	{
		//if(linked_list_ptr->id == -1) //avoid the dummy node.
		//{
		//	linked_list_ptr = linked_list_ptr->next;
		//}
		//sect_list = getAllSect(linked_list_ptr);
		sect_list = getVehSect(linked_list_ptr);  // 在一個讀取時間內，裡面是不會有重複的id的，相當於mapping的機制。
		/*
		if(flag_list == NULL){
			flag_list = sect_list;
			flag_list_tail = getListTail(sect_list);
		}else{
			flag_list_tail->next = sect_list;
			flag_list_tail = getListTail(sect_list);
		}
		*/
		if(flag_list == NULL){
			flag_list = sect_list;
			flag_list_tail = getListTail(flag_list);
		}else{
			ListForFlag(sect_list, flag_list, flag_list_tail);
			flag_list_tail = getListTail(flag_list);
		}
		list_size = getListSize(flag_list);
		/*
		printf("\nflag_list: %d\n", list_size);
		printList(flag_list);
		printf("\n");
		*/

		// Start of processing the Big map trasaction.
		Big_flag_list = ChangeInToBig(flag_list);

		/*
		printf("\nbig_flag_list: %d\n", getListSize(Big_flag_list));
		printList(Big_flag_list);
		printf("\n");
		*/


		//if(msg_num != getListSize(sect_list->next)) //skip the dummy node. msg_num should be equal to sect_list's size.
		if(msg_num != getListSize(sect_list)) //msg_num should be equal to sect_list's size.
		{
			fprintf(stderr, "***** ERROR: In carListTraverse(), the section list's size is not equal to the message number. *****\n");
			exit(1);
		}
		//sect_list_ptr = sect_list->next; //skip the dummy node.
		sect_list_ptr = sect_list;


		fprintf(fl_ptr, "\n");
		while(linked_list_ptr != NULL) //first list traverse.
		{

			linked_list_second_ptr = linked_list_ptr->second_list;

			while(linked_list_second_ptr != NULL) //second list traverse.
			{
				/* Get new small map index */
				//val_buf = IArrSearch(sect_mapping_tb, msg_num, linked_list_second_ptr->id); //search the mapped index.
				val_buf = ListSearch(flag_list, linked_list_second_ptr->id); //search the mapped index.
				
				/* Get new Big map index */
				org_big_target = GetBigMap(linked_list_second_ptr->id); // org_big_target is the original big map index
				// printf("org_big_target: %d", org_big_target);
				big_val_buff = ListSearch(Big_flag_list, org_big_target);


				if(val_buf == -1)
				{
					fprintf(stderr, "***** In carListTraverse(), there is a section id that hasn't be recorded in the sect_mapping_rb[] *****\n");
					exit(1);
				}
				fprintf(fl_ptr, "%d ", val_buf);
				fprintf(fl_output_map, "%d,%d ", val_buf, big_val_buff);
				//printf("%d ", sect_mapping_tb[val_buf]);
				//printf("%d, ", linked_list_second_ptr->id);
				linked_list_second_ptr = linked_list_second_ptr->next;
			}
			fprintf(fl_ptr, "G\n");
			
			linked_list_ptr = linked_list_ptr->next;
		}

		// freeList(sect_list);
		free(sect_mapping_tb);
	}
}

void listTraverse(struct linked_list *head) //traverse the linked list and display their elements.
{
	struct linked_list *linked_list_ptr = head;

	if(head == NULL)
	{
		printf("In listTraverse(), head is NULL. No node is printed\n");
	}
	else
	{
		printf("\n");
		while(linked_list_ptr != NULL)
		{
			printf("id: %d, val: %d, name: %s, x: %lf, y:%lf .\n", linked_list_ptr->id, linked_list_ptr->val, linked_list_ptr->name, linked_list_ptr->x, linked_list_ptr->y);
			linked_list_ptr = linked_list_ptr->next;
		}
	}
}

int getListSize(struct linked_list *head) //return the size of a linked list.
{
	int size = 0;
	struct linked_list *linked_list_ptr = head;

	while(linked_list_ptr != NULL)
	{
		size ++;
		linked_list_ptr = linked_list_ptr->next;
	}

	return size;
}

void printList(struct linked_list *head)
{
	struct linked_list *linked_list_ptr = head;

	while(linked_list_ptr != NULL)
	{
		printf("%d ", linked_list_ptr->id);
		linked_list_ptr = linked_list_ptr->next;
	}
}

int ListSearch(struct linked_list *head, int target)
{
	struct linked_list *linked_list_head = head;
	struct linked_list *linked_list_ptr = head;
	int counter = 0;
	int start_bool =0;
	while(linked_list_ptr != NULL)
	{
		if (linked_list_ptr->id == target){
			return counter;
		}else{
			counter += 1;  // compare to the next index
		}
		linked_list_ptr = linked_list_ptr->next;
	}
}

int listRangeFindId(struct linked_list *head, struct linked_list* end, int target){
	struct linked_list *linked_list_ptr = head;
	int dupi_ = 0;
	while(linked_list_ptr != NULL)
	{
		if (linked_list_ptr->id == target){
			dupi_ = 1;
			return dupi_;
		}
		linked_list_ptr = linked_list_ptr->next;
		if (linked_list_ptr == end){
			break;
		}
	}
	return dupi_;
}

struct linked_list *getAllSect(struct linked_list *car_list) //insert all the sections into a new section list(without duplicate) from 'car_list', and return the section list's head.
{
	struct linked_list *car_list_ptr = car_list, *sect_list = NULL, *sect_list_tail = NULL, *sect_list_ptr = NULL; //sect_list_ptr is used to traverse the second_list of car_list_ptr.

	if(car_list == NULL)
	{
		fprintf(stderr, "***** In getAllSect(), the car_list is NULL. *****\n");
		exit(1);
	}

	sect_list = (struct linked_list *) malloc(sizeof(struct linked_list) * 1);
	sect_list->id = -1; //dummy node of section list.
	sect_list->second_list = NULL;
	sect_list->second_list_tail = NULL;
	sect_list->next = NULL;
	sect_list_tail = sect_list;

	while(car_list_ptr != NULL) //traverse all car. Having such complexity, I introspect myself......
	{
		sect_list_ptr = car_list_ptr->second_list->next; //skip the dummy node.
		while(sect_list_ptr != NULL) //traverse each car's passing section.
		{
			if(findListId(sect_list, sect_list_ptr->id) == NULL)
			{
				sect_list_tail = listInsert(sect_list_tail, sect_list_ptr->id);
			}

			sect_list_ptr = sect_list_ptr->next;
		}

		car_list_ptr = car_list_ptr->next;
	}

	return sect_list;
}

void outputData(int car_num, int msg_num, int t0, int rb_num, struct linked_list *car_list, struct linked_list *sect_list, FILE *fl_ptr, int time, int v_size) //output the suitable data for experiment into the 'fl_ptr'.
{
	struct linked_list *car_list_ptr = car_list, *sect_list_ptr = sect_list;
	//int sect_list_num = malloc(car_num * sizeof(int)); //record the message number each car should receive.
	int sect_list_num[car_num]; //record the message number each car should receive.
	int val_buf = 0, med = 0;
	int seed = 0;
	double float_buf = 0.0, sum = 0.0;
	int **car_to_msg_birth = NULL; //record the bv(m) of each car's each message.
	double **car_to_msg_priority = NULL; //record the pv(m) of each car's each message.

	memset(sect_list_num, 0, car_num * sizeof(int)); //initialize the array. NOTE: the array with dynamic allocation can't be initialized by the method arr[num] = {0}.
	car_to_msg_birth = (int **) malloc(sizeof(int *) * car_num);
	car_to_msg_priority = (double **) malloc(sizeof(double *) * car_num);

	if(car_list == NULL)
	{
		fprintf(fl_ptr, "In outputData(), car_list is NULL, so there is not any output data.\n");
		return;
	}

	//int slt = 0;
	//scanf("%d", &slt);
	//fprintf(fl_ptr, "%d\n%d\n%d\n%d\n%d\n", car_num, msg_num, t0, rb_num, slt); //some basic numbers of the data.
	fprintf(fl_ptr, "%d\n%d\n%d\n%d\n", car_num, msg_num, t0, rb_num); //some basic numbers of the data.

	// car_num: needed car,  msg_num: needed map(no reapeat),  t0: start time(own decided), rb_num: (own decided)
	
	//
	//
	//
	//
	//
	/*
	fprintf(fl_ptr, "\n");
	for(int i = 0; i < car_num; i ++) //the messages number a car should receive.
	{
		//printf("%d\n", sect_list_num[i] = getListSize(car_list_ptr->second_list->next)); //second_list skips the dummy node, and we record the messages number of each car.
		fprintf(fl_ptr, "%d\n", sect_list_num[i] = getListSize(car_list_ptr->second_list)); //record the messages number of each car.
		*(car_to_msg_birth + i) = (int *) malloc(sizeof(int) * sect_list_num[i]); //allocate the space for bv(m) according to the messages number of each car.
		*(car_to_msg_priority + i) = (double *) malloc(sizeof(double) * sect_list_num[i]); //allocate the space for pv(m) according to the messages number of each car.
		car_list_ptr = car_list_ptr->next;
		//printf("Here is the sect_list: %d\n",sect_list_num[i]);
	}
	*/
	// fl_ptr: ,  sect_list_num:
	// v_0: 1 (car0 needed map num)
	// v_1: 2 



	// v_0: m_1
	// v_1: m_2, m_1
	fprintf(fl_ptr, "Start: %d\n",iter_);
	iter_ += 1;
	carListTraverse(car_list, msg_num, fl_ptr);
	fprintf(fl_ptr, "End: \n");


	/*scanf("%d", &seed);
	while(seed == 0 || seed == -1)
	{
		fprintf(stderr, "Invalid value, ReEnter the seed(should not be 0 or -1): \n");
		scanf("%d", &seed);
	}*/

	// 
	//
	//
	//
	//
	/*
	car_list_ptr = car_list;
	sect_list_ptr = sect_list;
	med = findMedVal(sect_list_ptr, msg_num) + 1; //get the median of the repeatedly received number.
	srand(seed);
	for(int i = 0; i < car_num; i ++) //set bv(m).
	{
		//sect_list_ptr = car_list_ptr->second_list->next; //skip the dummy node.
		sect_list_ptr = car_list_ptr->second_list;
		for(int j = 0; j < sect_list_num[i]; j ++)
		{
			if(findListId(sect_list, sect_list_ptr->id)->val > med) //let the one who has larger repeatedly received number has smaller bv(m).
			{
				car_to_msg_birth[i][j] = t0 - ((rand() % BIRTH_INTERVAL) + 1); //plus one to avoid bv(m) = t0.
			}
			else 
			{
				car_to_msg_birth[i][j] = t0 - 2 * BIRTH_INTERVAL - ((rand() % BIRTH_INTERVAL) + 1); //set their bv(m) at the interval of t0 - 2*BIRTH_INTERVAL and t0 - 3*BIRTH_INTERVAL.
			}
			sect_list_ptr = sect_list_ptr->next;
		}
		car_list_ptr = car_list_ptr->next;
	}
	srand(0);
	//set pv(m).
	for(int i = 0; i < car_num; i ++) //set pv(m).
	{ 
		float_buf = 0.0;
		sum = 0.0;
		for(int j = 0; j < sect_list_num[i]; j ++) //we need the priority to disproportionate to their birth. So we first calculate their reciprocal, and then adjust them to make the sum of the reciprocal be 1 by multiplying the reciprocal of the unadjusted sum, e.g., (1 + 1/2 + 1/3) * 6/11 = 1.
		{
			car_to_msg_priority[i][j] = 1.0 / (double) car_to_msg_birth[i][j];
			float_buf = float_buf + car_to_msg_priority[i][j];
		}
		float_buf = 1.0 / float_buf; //calculate the reciprocal of the unadjusted sum.
		for(int j = 0; j < sect_list_num[i]; j ++)
		{
			car_to_msg_priority[i][j] = car_to_msg_priority[i][j] * float_buf;
			sum = sum + car_to_msg_priority[i][j];
		}

		if(sum > 1.0) //adjust each message priority of a vehicle to make the sum be exactly 1.
		{
			val_buf = rand() % sect_list_num[i];
			car_to_msg_priority[i][val_buf] = car_to_msg_priority[i][val_buf] - (sum - 1.0);
		}
		else if(sum < 1.0)
		{
			val_buf = rand() % sect_list_num[i];
			car_to_msg_priority[i][val_buf] = car_to_msg_priority[i][val_buf] + (1.0 - sum);
		}
	}

	// Build a 2-D array
	int **a = (int**)malloc(car_num*sizeof(int*));
	for (int i = 0; i < 2; i++) a[i] = (int*)malloc(2*sizeof(int)); 

	fprintf(fl_ptr, "\n");
	for(int i = 0; i < car_num; i ++) //display pv(m). priority
	{
		float_buf = 0.0;
		for(int j = 0; j < sect_list_num[i]; j ++)
		{
			if(car_to_msg_priority[i][j] > 1 || car_to_msg_priority[i][j] < 0)
			{
				fprintf(stderr, "***** ERROR: Priority of the message %d in car %d is an invalid value %lf. *****\n", j, i, car_to_msg_priority[i][j]);
				exit(1);
			}
			fprintf(fl_ptr, "%lf ", car_to_msg_priority[i][j]); // i: car, j: maps
			float_buf = float_buf + car_to_msg_priority[i][j];
		}
		fprintf(fl_ptr, "\n");
		if(float_buf - 1.0 > 0.00001)
		{
			fprintf(stderr, "***** ERROR: In outputData(), sum %lf of the priority of the messages in car %d is not equal to 1. *****\n", float_buf, i);
			exit(1);
		}
	}
	*/

	fprintf(fl_ptr, "\n");
	/*
	for(int i = 0; i < car_num; i ++) //display bv(m). birth time
	{
		int temp = 3 * BIRTH_INTERVAL;

		for(int j = 0; j < sect_list_num[i]; j ++)
		{
			if(car_to_msg_birth[i][j] >= t0 || car_to_msg_birth[i][j] < t0 - temp || car_to_msg_birth[i][j] <= 0) //temp = 3 * BIRTH_INTERVAL.
			{
				fprintf(stderr, "***** ERROR: In outputData(), Birth of the message %d in car %d is an invalid value %d. *****\n", j, i, car_to_msg_birth[i][j]);
				exit(1);
			}
			fprintf(fl_ptr, "%d ", car_to_msg_birth[i][j]);
		}
		fprintf(fl_ptr, "\n");
	}*/
}

int IArrSearch(int *arr, int arr_size, int target) //search the target in the integer array, and return it's index or -1 if the target is not found.
{
	for(int i = 0; i < arr_size; i ++)
	{
		if(*(arr + i) == target)  // target is the original index
		{

			return i; // i is the new index
		}
	}

	return -1;
}

struct linked_list *getListTail(struct linked_list *head) //return the tail of that linked list.
{
	struct linked_list *linked_list_ptr = head;

	if(linked_list_ptr != NULL)
	{
		while(linked_list_ptr->next != NULL) //the signature of the tail is that its next node is NULL.
		{
			linked_list_ptr = linked_list_ptr->next;
		}
	}

	return linked_list_ptr;
}

int vCntrCheck(struct linked_list *car_list, int car_num, struct linked_list *sect_list, int msg_num) //check wether the repeatedly received number is correct. return 0 if it's not correct.
{
	struct linked_list *car_list_ptr = NULL, *sect_list_ptr = sect_list;
	int cntr = 0;

	for(int i = 0; i < msg_num; i ++)
	{
		car_list_ptr = car_list;
		cntr = 0;
		for(int j = 0; j < car_num; j ++)
		{
			if(findListId(car_list_ptr->second_list->next, sect_list_ptr->id) != NULL)
			{
				cntr ++;
			}
			car_list_ptr = car_list_ptr->next;
		}
		if(cntr != sect_list_ptr->val)
		{
			fprintf(stderr, "***** ERROR: In cCntrCheck(), the repeatedly received number %d of message %d doesn't match the cntr %d. *****\n", sect_list_ptr->val, sect_list_ptr->id, cntr);
			return 0;
		}
		sect_list_ptr = sect_list_ptr->next;
	}

	return 1;
}

int findMedVal(struct linked_list *head, int msg_num) //find and return the median val in the linked list.
{
	int med = 0;
	int *val_arr = NULL; //sort the array first, and then select the median.
	struct linked_list *list_ptr = head;

	if(list_ptr == NULL) //value check.
	{
		fprintf(stderr, "***** ERROR: In findMidVal(), head or head->next is NULL. *****");
		exit(1);
	}
	else if(list_ptr->id == -1)
	{
		list_ptr = list_ptr->next;
		if(list_ptr == NULL)
		{
			fprintf(stderr, "***** ERROR: In findMidVal(), head->next is NULL. *****");
			exit(1);
		}
	}

	val_arr = (int *) malloc(sizeof(int) * msg_num);
	for(int i = 0; i < msg_num; i ++) //copy the val into array for the sorting.
	{
		*(val_arr + i) = list_ptr->val;
		list_ptr = list_ptr->next;
	}
	qsort(val_arr, msg_num, sizeof(int), valCmp); //sorting

	med = val_arr[msg_num / 2];

	free(val_arr);

	return med;
}

int valCmp(const void *val1, const void *val2) //qsort cmp function.
{
	return *((int *)val1) - *((int *) val2);
}

int toVehNumTime(FILE *fl_ptr, int veh_num) //let the file pointer points to the time which has vehicle number bigger than 'vch_num'. Return 0 if failed.
{
	char line[1024] = "\0";
	char *ptr = NULL;
	int veh_num_cntr = 0;

	while((ptr = fgets(line, 1024, fl_ptr)) != NULL) //fetch the lines.
	{
		if(strstr(ptr, "<timestep") != NULL) //located at <timestep>
		{
			veh_num_cntr = 0;
			while((ptr = fgets(line, 1024, fl_ptr)) != NULL) //keep counting until occures </timestep>
			{
				if(strstr(ptr, "</timestep") != NULL) //go to next timestep when this time is end but the number hasn't met.
				{
					break;
				}

				veh_num_cntr ++;
			}

			if(veh_num_cntr > veh_num)
			{
				return 1;
			}
		}
	}

	return 0;
}

char *parseVehName(char *line, char *veh_name) //parse the vehicle's name from a xml line. Return the name if success, else NULL.
{
	char *ptr = NULL, *copy_ptr = veh_name; //copy_ptr is used when copy the character from line into veh_name.
	char *line_end = line + strlen(line);
//sample of the vehicle tuple: <vehicle id="randUni777:1" x="7725.29" y="10326.10" angle="155.12" type="passenger2a" speed="15.65" pos="4.00" lane=":-12184_6_0" slope="0.00"/>.
	ptr = strstr(line, "<vehicle id=\""); //find the signature of the format of the vehicle's name, and move ptr to the beginning of that format.
	if(ptr != NULL && copy_ptr != NULL)
	{
		*copy_ptr = '\0';
		ptr = ptr + strlen("<vehicle id=\""); //move ptr to the beginning of the name.
		while(*ptr != '\"' && ptr < line_end) //copy the name into veh_name.
		{
			*copy_ptr ++ = *ptr ++;
		}
		*copy_ptr = '\0';
		
		return veh_name;
	}

	return NULL;
}

double parseVehX(char *line) //parse the vehicle's x coordinate. Return CORD_MAX if failed.
{
	char *ptr = NULL;
	
	ptr = strstr(line, "x=\""); //find the signature of the format of the vehicle's x coordinate, and move ptr to it.
	if(ptr != NULL)
	{
		ptr = ptr + strlen("x=\"");
		return atof(ptr); //covert ascii digits into double until it met the ascii that is not a digits.
	}

	return CORD_MAX;
}

double parseVehY(char *line) //parse the vehicle's y coordinate. Return CORD_MAX if failed.
{
	char *ptr = NULL;

	ptr = strstr(line, "y=\""); //find the signature of the format of the vehicle's y coordinate, and move ptr to it.
	if(ptr != NULL)
	{
		ptr = ptr + strlen("y=\"");
		return atof(ptr); //covert ascii digits into double until it met the ascii that is not a digits.
	}

	return CORD_MAX;
}

struct linked_list *listNameInsert(struct linked_list **head, struct linked_list *tail, char *name) //insert a new node with name. Return the tail.
{
	if(tail != NULL) //common case.
	{
		tail->next = (struct linked_list *) malloc(sizeof(struct linked_list) * 1);
		tail = tail->next; //move the tail.
	}
	else if(*head == NULL) //head is NULL.
	{
		*head = (struct linked_list *) malloc(sizeof(struct linked_list) * 1);
		tail = *head;
	}
	else  if(tail == NULL) //tail is NULL.
	{
		tail = (struct linked_list *) malloc(sizeof(struct linked_list) * 1);
	}
	tail->id = -1;
	tail->val = 0;
	tail->x = 0.0;
	tail->y = 0.0;
	tail->second_list = NULL;
	tail->second_list_tail = NULL;
	tail->next = NULL;

	strcpy(tail->name, name);

	return tail;
}

int GetBigMap(int i){

	int loc_temp = i / 50;
    int line_temp = i / 100;
    int loc = i - 50 * loc_temp;
    int map_loc = loc / 2;
    int ans = map_loc + 25 * line_temp;

    return ans;
}

struct linked_list *ChangeInToBig(struct linked_list *sect_list){
	struct linked_list *sect_head = NULL, *sect_tail = NULL; //let the id in linked_list be the section id.
	struct linked_list *v_ptr = sect_list, *s_ptr = NULL;
	int sect = 0;

	while(v_ptr != NULL) 
	{
		sect = v_ptr->id;  // v_ptr = flag_list
		// Change to Big map index
		sect = GetBigMap(sect);
		if((s_ptr = findListId(sect_head, sect)) == NULL)  // NOT duplicate then save
		{
			sect_tail = listIdInsert(&sect_head, sect_tail, sect);
		}
		v_ptr = v_ptr->next;
	}

	return sect_head;
}

void *ListForFlag(struct linked_list *veh_list, struct linked_list *flag_list_head, struct linked_list *flag_list_tail ){
	struct linked_list *v_ptr = veh_list, *s_ptr = NULL;
	struct linked_list *snd_ptr = NULL;
	int sect = 0;

	while(v_ptr != NULL) //iterate in vehicle linked list.
	{
		sect = v_ptr->id;
		if((s_ptr = findListId(flag_list_head, sect)) == NULL)  // NOT duplicate then save
		{
			flag_list_tail = listIdInsert(&flag_list_head, flag_list_tail, sect);
		}
		v_ptr = v_ptr->next;
	}

} //extract all the sections vehicles passed. Return the head of the section.

struct linked_list *getVehSect(struct linked_list *veh_list) //extract all the sections vehicles passed. Return the head of the section.
{
	struct linked_list *sect_head = NULL, *sect_tail = NULL; //let the id in linked_list be the section id.
	struct linked_list *v_ptr = veh_list, *s_ptr = NULL;
	struct linked_list *snd_ptr = NULL;
	int sect = 0;
	int sect_a_row = (X_END - X_START) / 10;

	while(v_ptr != NULL) //iterate in vehicle linked list.
	{
		snd_ptr = v_ptr->second_list;
		while(snd_ptr != NULL) //iterate in the section list of each vehicle.
		{
			sect = snd_ptr->id;
			if((s_ptr = findListId(sect_head, sect)) == NULL)  // NOT duplicate then save
			{
				sect_tail = listIdInsert(&sect_head, sect_tail, sect);
				sect_tail->val ++;
			}
			else
			{
				s_ptr->val ++;
			}

			snd_ptr = snd_ptr->next;
		}

		v_ptr = v_ptr->next;
	}

	return sect_head;
}

struct linked_list *listIdInsert(struct linked_list **head, struct linked_list *tail, int id) //insert a new node with id. Return the tail.
{
	if(tail != NULL) //common case.
	{
		tail->next = (struct linked_list *) malloc(sizeof(struct linked_list) * 1);
		tail = tail->next; //move the tail.
	}
	else if(*head == NULL) //head is NULL.
	{
		*head = (struct linked_list *) malloc(sizeof(struct linked_list) * 1);
		tail = *head;
	}
	else  if(tail == NULL) //tail is NULL.
	{
		tail = (struct linked_list *) malloc(sizeof(struct linked_list) * 1);
	}
	tail->id = id;
	tail->val = 0;
	tail->x = 0.0;
	tail->y = 0.0;
	tail->name[0] = '\0';
	tail->second_list = NULL;
	tail->second_list_tail = NULL;
	tail->next = NULL;

	return tail;
}

int listNameErrCheck(struct linked_list *head) //check whether the name is duplicate. Return 0 if error.
{
	struct linked_list *ptr = head;
	int list_size = getListSize(head);
	char **name_arr = NULL;

	name_arr = (char **) malloc(sizeof(char *) * list_size); //construct the array of all names.
	for(int i = 0; i < list_size; i ++)
	{
		*(name_arr + i) = (char *) malloc(sizeof(char) * 64);
		strcpy(*(name_arr + i), ptr->name);
		ptr = ptr->next;
	}

	for(int i = 0; i < list_size; i ++) //check each name in the head.
	{
		for(int j = 0; j < list_size; j ++) //check whether there is duplication by comparing the name with all names except itself.
		{
			if(j != i)
			{
				if(strcmp(*(name_arr + i), *(name_arr + j)) == 0) //comparing
				{
					fprintf(stderr, "name[%d] %s, and name[%d] %s are the same\n", i, *(name_arr + i), j, *(name_arr + j));
					return 0;
				}
			}
		}
	}

	return 1;
}

int listIdErrCheck(struct linked_list *head) //check whether the id is duplicate. Return 0 if error.
{
	struct linked_list *ptr = head;
	int list_size = getListSize(head);
	int *id_arr = NULL;

	id_arr = (int *) malloc(sizeof(int) * list_size); //construct the array of all ids.
	for(int i = 0; i < list_size; i ++)
	{
		*(id_arr + i) = ptr->id;
		ptr = ptr->next;
	}

	for(int i = 0; i < list_size; i ++) //check each id in the head.
	{
		for(int j = 0; j < list_size; j ++) //check whether there is duplication by comparing the id with all id except itself.
		{
			if(j != i)
			{
				if(*(id_arr + i) == *(id_arr + j)) //comparing
				{
					fprintf(stderr, "id[%d] %d, and id[%d] %d are the same\n", i, *(id_arr + i), j, *(id_arr + j));
					return 0;
				}
			}
		}
	}

	return 1;
}

void doubleListDisplay(struct linked_list *list_head) //print the content in a linked list doubly.
{
	int first_num = getListSize(list_head), second_num = 0;
	struct linked_list *first_ptr = list_head, *second_ptr = NULL;

	printf("\n");
	for(int i = 0; i < first_num; i ++)
	{
		printf("First layer list id: %d, val: %d, name: %s, x: %lf, y: %lf\n", first_ptr->id, first_ptr->val, first_ptr->name, first_ptr->x, first_ptr->y);

		second_ptr = first_ptr->second_list;
		second_num = getListSize(second_ptr);
		for(int j = 0; j < second_num; j ++)
		{
			printf("\tSecond layer list id: %d, val: %d, name: %s, x: %lf, y: %lf\n", second_ptr->id, second_ptr->val, second_ptr->name, second_ptr->x, second_ptr->y);
			second_ptr = second_ptr->next;
		}

		first_ptr = first_ptr->next;
	}
}

struct linked_list *parseVehSect(FILE *fl_ptr, struct linked_list *veh_head, int time) //parse the section that the vehicles will pass in next 'time' seconds. Return vehicle linked list if success, else NULL.
{
	char line[1024] = "\0", name[64] = "\0";
	char *ptr = NULL, *name_ptr = NULL;
	double x = 0.0, y = 0.0;
	int sect = 0;
	int sect_a_row = (X_END - X_START) / 10;
	struct linked_list *veh_ptr = NULL;

	for(int i = 0; i < time; i ++) //times iteration.
	{
		if(toNextTimeStep(fl_ptr, line) == NULL) //when fl_ptr meet the EOF, return NULL.
		{
			return NULL;
		}

		while((ptr = fgets(line, 1024, fl_ptr)) != NULL) //parse the section that the existed vehicles will pass.
		{
			if(strstr(ptr, "</timestep>") != NULL) //when meet the end of the time step, holds the parsing.
			{
				break;
			}

			name_ptr = parseVehName(ptr, name);
			if((veh_ptr = findListName(veh_head, name_ptr)) != NULL) //parse the passing section only when the vehicle exists.
			{
				x = parseVehX(ptr);
				y = parseVehY(ptr);
				sect = ((int) (x - X_START) / 10) + sect_a_row * (int) ((y - Y_START) / 10); //calculate the section id.
				// printf("%d,x: %lf, X_START: %d, y: %lf, Y_Start: %d\n", sect, x, X_START, y, Y_START);
				if(findListId(veh_ptr->second_list, sect) == NULL)
				{
					veh_ptr->second_list_tail = listIdInsert(&(veh_ptr->second_list), veh_ptr->second_list_tail, sect); //insert the sect into veh node.
				}
			}
		}
	} //end of the times iteration.

	return veh_head;
}
char *toNextTimeStep(FILE *fl_ptr, char *line) //move file pointer to the beginning of next time step. Return the next time stpe line if success, else NULL.
{
	char *ptr = NULL;

	while((ptr = fgets(line, 1024, fl_ptr)) != NULL) //get lines.
	{
		const char* d = "\"";
		char *p;
		p = strtok(ptr, d);
		p = strtok(NULL, d);
		time_val = atoi(p);
		if(time_val > last_time){
			// printf("%s ",p);
			return line;
		}
	}

	return NULL;
}

struct linked_list *findListName(struct linked_list *head, char *name) //find the 'name' in the list. Return the list if success, else NULL.
{
	struct linked_list *ptr = head;

	while(ptr != NULL) //list iteration.
	{
		if(strcmp(ptr->name, name) == 0) //return the list when its name is found.
		{
			return ptr;
		}

		ptr = ptr->next;
	}

	return NULL;
}

void setRSUCORD() //set the coordinate of the RSUs by reading file RSU_CORD.txt.
{
	char line[1024] = "\0";
	FILE *RSU_fl = NULL;
	int x = 0, y = 0;

	RSU_fl = fopen("./RSU_CORD.txt", "r");
	if(RSU_fl == NULL)
	{
		fprintf(stderr, "***** ERROR: File RSU_CORD.txt is not found in current directory. *****\n");
		exit(1);
	}

	for(int i = 0; i < RSU_NUM; i ++)
	{
		if(fgets(line, 1024, RSU_fl) != NULL)
		{
			sscanf(line, "%d%d", &x, &y);
			RSU_CORD[i][0] = x;
			RSU_CORD[i][1] = y;
		}
		else
		{
			fprintf(stderr, "***** ERROR: File RSU_CORD.txt has not enough coordinate. *****\n");
			exit(1);
		}
	}

	X_START = RSU_CORD[0][0] - 500;
	X_END = RSU_CORD[0][0] + 500;
	Y_START = RSU_CORD[0][1] - 500;
	Y_END = RSU_CORD[0][1] + 500;

	//for(int i = 0; i < RSU_NUM; i ++)
	//{
	//	printf("%d %d\n", RSU_CORD[i][0], RSU_CORD[i][1]);
	//}

	fclose(RSU_fl);
}

struct linked_list *getVehInTRSU(FILE *fl_ptr, int flag) //parse the vehicles in RSU covered range at a time step, i.e., parse until meet </timestep>, from 'fl_ptr', and record the passed sections as well. Return the linked list's head of the vehicles if success, else NULL.
{
	struct linked_list *veh_head = NULL, *veh_tail = NULL;
	char line[1024] = "\0";
	char name[64] = "\0";
	char *ptr = NULL;
	double x = 0.0, y = 0.0;
	int sect = 0;
	int sect_a_row = (X_END - X_START) / 10;
	int radXrad = RSU_RAD * RSU_RAD;
	int dist = 0;

	while((ptr = fgets(line, 1024, fl_ptr)) != NULL) //get lines.
	{
		if(strstr(ptr, "</timestep>") != NULL) //return the vehicle's list when encounter the end of of time setp.
		{
			return veh_head;
		}

		if(parseVehName(line, name) != NULL) //when we find the vehicle, check for its coordinate, and decide whether should insert it or not.
		{
			x = parseVehX(line);
			y = parseVehY(line);
			dist = (int) (x - RSU_CORD[0][0]) * (int) (x - RSU_CORD[0][0]) + (int) (y - RSU_CORD[0][1]) * (int) (y - RSU_CORD[0][1]); //calculate the distance from vehicle to RSU.
			if(dist < radXrad) //insert when the vehicle is covered by RSU.
			{
				veh_tail = listNameInsert(&veh_head, veh_tail, name);
				veh_tail->x = x;
				veh_tail->y = y;
				sect = ((int) (veh_tail->x - X_START) / 10) + sect_a_row * (int) ((veh_tail->y - Y_START) / 10); //calculate the section id.
				veh_tail->second_list_tail = listIdInsert(&(veh_tail->second_list), veh_tail->second_list_tail, sect); //insert the sect into veh node.
			}
		}
	}

	return NULL;
}

struct linked_list *filterList(struct linked_list *head, int target_size) //filter the list randomly into 'target_size', if the list size is already lesser than 'target_size' , then it won't be filtered. Return the list.
{
	int org_size = getListSize(head);
	int tmp = 0, d_size = 0;
	struct linked_list *ptr = head, *d_ptr = NULL;

	srand(-1);
	if(org_size > target_size) //only the list which has redundant node need to be filtered.
	{
		d_size = org_size - target_size; //the number of nodes should be deleted.

		for(int itr = 0; itr < d_size; itr ++)
		{
			d_ptr = getIdxNode(ptr, (rand() % (org_size - itr))); //get the node that should be deleted.
			ptr = deleteNode(ptr, d_ptr);
		}
	}

	return ptr;
}

struct linked_list *getIdxNode(struct linked_list *head, int idx) //get the node that index is 'idx' in the list. Return the list ptr if success, else NULL.
{
	int list_size = getListSize(head);
	struct linked_list *ptr = head;
	
	if(list_size <= idx)
	{
		return NULL;
	}

	for(int itr = 0; itr < idx; itr ++)
	{
		ptr = ptr->next;
	}

	return ptr;
}